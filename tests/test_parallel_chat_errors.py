"""Tests for parallel_chat error handling."""

import pytest
from pydantic import BaseModel

from chatlas import (
    ChatOpenAI,
    parallel_chat,
    parallel_chat_structured,
    parallel_chat_text,
)

from ._test_providers import TestChatOpenAI


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_error_return_mode():
    """Test that on_error='return' stops new requests but completes in-flight ones."""
    # Use an invalid model to trigger an error - use explicit dummy key since not valid VCR test
    chat = ChatOpenAI(api_key="test", model="gpt-4-does-not-exist-12345")

    prompts = [
        "Say 'A'",
        "Say 'B'",
        "Say 'C'",
    ]

    # With max_active=1, only the first request should attempt and fail
    results = await parallel_chat(
        chat,
        prompts,
        max_active=1,
        on_error="return",
    )

    # First should be an exception, others should be None (not submitted)
    assert len(results) == 3
    assert isinstance(results[0], Exception)
    assert results[1] is None  # Not submitted
    assert results[2] is None  # Not submitted


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_error_continue_mode():
    """Test that on_error='continue' processes all requests despite errors."""
    # Use an invalid model to trigger an error - use explicit dummy key since not valid VCR test
    chat = ChatOpenAI(api_key="test", model="gpt-4-does-not-exist-12345")

    prompts = [
        "Say 'A'",
        "Say 'B'",
        "Say 'C'",
    ]

    # With max_active=1, only the first request should attempt and fail
    results = await parallel_chat(
        chat,
        prompts,
        max_active=1,
        on_error="continue",
    )
    # All should be exceptions
    assert len(results) == 3
    assert all(isinstance(r, Exception) for r in results)


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_error_stop_mode():
    """Test that on_error='stop' raises immediately on first error."""
    # Use an invalid model to trigger an error - use explicit dummy key since not valid VCR test
    chat = ChatOpenAI(api_key="test", model="gpt-4-invalid-12345")

    prompts = [
        "Say 'A'",
        "Say 'B'",
        "Say 'C'",
    ]

    # Should raise an exception immediately
    with pytest.raises(Exception):
        await parallel_chat(
            chat,
            prompts,
            max_active=1,
            on_error="stop",
        )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_text_error_handling():
    """Test that parallel_chat_text handles errors correctly."""
    # Use explicit dummy key since testing error handling with invalid model
    chat = ChatOpenAI(api_key="test", model="gpt-4-nonexistent-99999")

    prompts = ["Say 'Hello'", "Say 'World'"]

    results = await parallel_chat_text(
        chat,
        prompts,
        max_active=1,
        on_error="return",
    )

    # Should return None for errored prompts
    assert len(results) == 2
    assert isinstance(results[0], Exception)  # First one errors
    # Second one may or may not run depending on timing


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_structured_error_handling():
    """Test that parallel_chat_structured handles errors correctly."""

    class Person(BaseModel):
        name: str
        age: int

    # Use explicit dummy key since testing error handling with invalid model
    chat = ChatOpenAI(api_key="test", model="invalid-model-12345")

    prompts = ["John, age 25", "Jane, age 30"]

    results = await parallel_chat_structured(
        chat,
        prompts,
        Person,
        max_active=1,
        on_error="return",
    )

    # Should have exceptions in results
    assert len(results) == 2
    assert isinstance(results[0], Exception)


@pytest.mark.asyncio
async def test_parallel_chat_empty_prompts_with_error_handling():
    """Test that empty prompts list works with error handling."""
    chat = TestChatOpenAI()

    results = await parallel_chat(chat, [], on_error="return")
    assert results == []

    results = await parallel_chat(chat, [], on_error="continue")
    assert results == []

    results = await parallel_chat(chat, [], on_error="stop")
    assert results == []

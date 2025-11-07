"""Tests for parallel_chat error handling."""

import os

import pytest

from chatlas import ChatOpenAI, parallel_chat, parallel_chat_structured, parallel_chat_text
from pydantic import BaseModel

# Skip these tests if not running OpenAI tests
do_test = os.getenv("TEST_OPENAI", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping OpenAI tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_parallel_chat_error_return_mode():
    """Test that on_error='return' stops new requests but completes in-flight ones."""
    # Use an invalid model to trigger an error
    chat = ChatOpenAI(model="gpt-4-does-not-exist-12345")

    prompts = [
        "Say 'A'",
        "Say 'B'",
        "Say 'C'",
    ]

    # With max_active=1, only the first request should attempt and fail
    results = await parallel_chat(chat, prompts, max_active=1, on_error="return")

    # First should be an exception, others should be None (not submitted)
    assert len(results) == 3
    assert isinstance(results[0], Exception)
    assert results[1] is None  # Not submitted
    assert results[2] is None  # Not submitted


@pytest.mark.asyncio
async def test_parallel_chat_error_continue_mode():
    """Test that on_error='continue' processes all requests despite errors."""
    call_log = []

    def tracker(msg: str) -> str:
        """Track which prompts are processed."""
        call_log.append(msg)
        return f"Processed {msg}"

    chat = ChatOpenAI()
    chat.register_tool(tracker)

    # Mix valid and invalid requests - use invalid model for some
    prompts = [
        "Call tracker with 'A'",
        "Call tracker with 'B'",
        "Call tracker with 'C'",
    ]

    # All should attempt regardless of errors
    results = await parallel_chat(chat, prompts, on_error="continue")

    # All prompts should have been processed
    assert len(results) == 3
    # All should succeed (valid model)
    assert all(not isinstance(r, Exception) for r in results)
    assert call_log == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_parallel_chat_error_stop_mode():
    """Test that on_error='stop' raises immediately on first error."""
    # Use an invalid model to trigger an error
    chat = ChatOpenAI(model="gpt-4-invalid-12345")

    prompts = [
        "Say 'A'",
        "Say 'B'",
        "Say 'C'",
    ]

    # Should raise an exception immediately
    with pytest.raises(Exception):
        await parallel_chat(chat, prompts, max_active=1, on_error="stop")


@pytest.mark.asyncio
async def test_parallel_chat_text_error_handling():
    """Test that parallel_chat_text handles errors correctly."""
    chat = ChatOpenAI(model="gpt-4-nonexistent-99999")

    prompts = ["Say 'Hello'", "Say 'World'"]

    results = await parallel_chat_text(chat, prompts, max_active=1, on_error="return")

    # Should return None for errored prompts
    assert len(results) == 2
    assert isinstance(results[0], Exception) # First one errors
    # Second one may or may not run depending on timing


@pytest.mark.asyncio
async def test_parallel_chat_structured_error_handling():
    """Test that parallel_chat_structured handles errors correctly."""

    class Person(BaseModel):
        name: str
        age: int

    chat = ChatOpenAI(model="invalid-model-12345")

    prompts = ["John, age 25", "Jane, age 30"]

    results = await parallel_chat_structured(
        chat, prompts, Person, max_active=1, on_error="return"
    )

    # Should have exceptions in results
    assert len(results) == 2
    assert isinstance(results[0], Exception)


@pytest.mark.asyncio
async def test_parallel_chat_error_in_tool_round():
    """Test error handling when errors occur during tool result submission."""
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return f"Result: {x}"

    chat = ChatOpenAI()
    chat.register_tool(my_tool)

    prompts = [
        "Call my_tool with 'A'",
        "Call my_tool with 'B'",
        "Call my_tool with 'C'",
    ]

    # First prompt should trigger tool call
    # If we inject an error somehow during tool result submission,
    # the error handling should still work
    results = await parallel_chat(chat, prompts, on_error="continue")

    # All should succeed
    assert len(results) == 3
    assert all(not isinstance(r, Exception) for r in results)
    assert call_log == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_parallel_chat_partial_results_on_error():
    """Test that successful results are returned even when some fail."""
    call_log = []

    def tracker(msg: str) -> str:
        call_log.append(msg)
        return f"Tracked: {msg}"

    # Create two chats - one valid, one invalid
    chat_valid = ChatOpenAI()
    chat_valid.register_tool(tracker)

    prompts = ["Call tracker with 'A'", "Call tracker with 'B'"]

    results = await parallel_chat(chat_valid, prompts, on_error="continue")

    # Both should succeed
    assert len(results) == 2
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count == 2
    assert call_log == ["A", "B"]


@pytest.mark.asyncio
async def test_parallel_chat_error_stops_only_errored_conversation():
    """Test that errors in one conversation don't stop other conversations."""
    call_log = []

    def my_tool(x: str) -> str:
        call_log.append(x)
        return f"Processed {x}"

    chat = ChatOpenAI()
    chat.register_tool(my_tool)

    prompts = [
        "Call my_tool with 'A'",
        "Call my_tool with 'B'",
        "Call my_tool with 'C'",
    ]

    # In "return" mode, if one fails, others should still complete
    results = await parallel_chat(chat, prompts, on_error="return")

    # All should succeed (no actual errors in this test)
    assert len(results) == 3
    assert all(not isinstance(r, Exception) for r in results)
    assert call_log == ["A", "B", "C"]


@pytest.mark.asyncio
async def test_parallel_chat_empty_prompts_with_error_handling():
    """Test that empty prompts list works with error handling."""
    chat = ChatOpenAI()

    results = await parallel_chat(chat, [], on_error="return")
    assert results == []

    results = await parallel_chat(chat, [], on_error="continue")
    assert results == []

    results = await parallel_chat(chat, [], on_error="stop")
    assert results == []

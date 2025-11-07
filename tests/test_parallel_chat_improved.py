"""Tests for improved parallel_chat with multi-turn tool support."""

import os

import pytest
from pydantic import BaseModel

from chatlas import ChatOpenAI, parallel_chat, parallel_chat_structured

# Skip these tests if not running OpenAI tests
do_test = os.getenv("TEST_OPENAI", "true")
if do_test.lower() == "false":
    pytest.skip("Skipping OpenAI tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_parallel_chat_multi_turn_tools():
    """Test that tools can trigger multiple LLM rounds."""
    call_log = []

    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        call_log.append(f"weather:{location}")
        return f"Weather in {location}: Sunny, 72°F"

    def get_time(location: str) -> str:
        """Get the current time for a location."""
        call_log.append(f"time:{location}")
        return f"Time in {location}: 3:00 PM"

    chat = ChatOpenAI(system_prompt="Be terse.")
    chat.register_tool(get_weather)
    chat.register_tool(get_time)

    # Prompt that should trigger both tools, then respond
    prompts = [
        "Get the weather in Seattle, then tell me the time there. Reply with both results.",
        "Get the weather in Tokyo, then tell me the time there. Reply with both results.",
    ]

    chats = await parallel_chat(chat, prompts)

    # Tools should have been called in order: Seattle tools, then Tokyo tools
    assert call_log == [
        "weather:Seattle",
        "time:Seattle",
        "weather:Tokyo",
        "time:Tokyo",
    ], f"Expected ordered tool calls, got {call_log}"

    # Each chat should have multiple turns
    assert len(chats) == 2
    # At least: user, assistant (with tools), user (tool results), assistant (final response)
    assert len(chats[0].get_turns()) >= 4, f"Expected at least 4 turns, got {len(chats[0].get_turns())}"
    assert len(chats[1].get_turns()) >= 4, f"Expected at least 4 turns, got {len(chats[1].get_turns())}"


@pytest.mark.asyncio
async def test_parallel_chat_ordering_preserved():
    """Test that tool execution order matches submission order."""
    execution_order = []

    def record(msg: str) -> str:
        """Record a message."""
        execution_order.append(msg)
        return f"Recorded: {msg}"

    chat = ChatOpenAI(system_prompt="Be terse. Always use the record tool as requested.")
    chat.register_tool(record)

    prompts = [
        "Call record with 'A'",
        "Call record with 'B'",
        "Call record with 'C'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Verify strict ordering
    assert execution_order == ["A", "B", "C"], f"Expected ['A', 'B', 'C'], got {execution_order}"
    assert len(chats) == 3


@pytest.mark.asyncio
async def test_parallel_chat_mixed_tools_and_no_tools():
    """Test conversations with and without tools."""
    call_log = []

    def helper() -> str:
        """A helper tool."""
        call_log.append("called")
        return "done"

    chat = ChatOpenAI(system_prompt="Be terse.")
    chat.register_tool(helper)

    prompts = [
        "Call the helper tool and tell me the result",
        "Just say 'hello' (don't use tools)",
        "Call the helper tool again and tell me the result",
    ]

    chats = await parallel_chat(chat, prompts)

    # Only prompts 0 and 2 should have used the tool
    assert call_log == ["called", "called"], f"Expected ['called', 'called'], got {call_log}"
    assert len(chats) == 3

    # Verify the middle chat didn't use tools
    turns = chats[1].get_turns()
    # Should be just: user, assistant (no tool calls in between)
    assert len(turns) == 2, f"Expected 2 turns for no-tool chat, got {len(turns)}"


@pytest.mark.asyncio
async def test_parallel_chat_chained_tool_calls():
    """Test that a conversation can have multiple rounds of tool calls."""
    call_log = []

    def step_one() -> str:
        """First step."""
        call_log.append("step_1")
        return "Step 1 complete"

    def step_two() -> str:
        """Second step."""
        call_log.append("step_2")
        return "Step 2 complete"

    chat = ChatOpenAI(system_prompt="Be terse. Follow instructions exactly.")
    chat.register_tool(step_one)
    chat.register_tool(step_two)

    prompts = [
        "First call step_one, wait for the result, then call step_two, and tell me when both are done",
        "Just call step_one and tell me when it's done",
    ]

    chats = await parallel_chat(chat, prompts)

    # First prompt should call both tools in sequence
    # Second prompt should only call step_one
    assert len(call_log) >= 3, f"Expected at least 3 tool calls, got {len(call_log)}: {call_log}"

    # Find indices
    step_1_indices = [i for i, x in enumerate(call_log) if x == "step_1"]
    step_2_indices = [i for i, x in enumerate(call_log) if x == "step_2"]

    # There should be 2 step_1 calls and at least 1 step_2 call
    assert len(step_1_indices) == 2, f"Expected 2 step_1 calls, got {len(step_1_indices)}: {call_log}"
    assert len(step_2_indices) >= 1, f"Expected at least 1 step_2 call, got {len(step_2_indices)}: {call_log}"

    # Within each round, tools should be processed in order
    # But between rounds, conversations can diverge (some may need more tools than others)
    # The key is that the first step_1 should happen before the second step_1 within the same round
    # Since both prompts start with step_one, both should call step_1 in round 1 (in order)
    # Then in round 2, the first prompt calls step_2
    # This test just verifies that multi-round tool calling works correctly

    assert len(chats) == 2


@pytest.mark.asyncio
async def test_parallel_chat_no_tools_registered():
    """Test that parallel_chat works normally when no tools are registered."""
    chat = ChatOpenAI(system_prompt="Be terse.")

    prompts = [
        "Say 'Alpha'",
        "Say 'Beta'",
        "Say 'Gamma'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Should complete successfully without any tool execution
    assert len(chats) == 3
    for c in chats:
        last_turn = c.get_last_turn()
        assert last_turn is not None
        assert last_turn.text is not None
        # Each chat should have exactly 2 turns (user prompt, assistant response)
        assert len(c.get_turns()) == 2


@pytest.mark.asyncio
async def test_parallel_chat_tool_ordering_with_rate_limiting():
    """Test that tool ordering is preserved even with rate limiting."""
    execution_order = []

    def record(id: str) -> str:
        """Record execution."""
        execution_order.append(id)
        return f"Recorded {id}"

    chat = ChatOpenAI(system_prompt="Be terse. Always use the record tool as requested.")
    chat.register_tool(record)

    # Use a larger set to test rate limiting
    prompts = [f"Call record with '{i}'" for i in range(10)]

    chats = await parallel_chat(chat, prompts, max_active=3, rpm=100)

    # Verify strict ordering despite rate limiting
    expected = [str(i) for i in range(10)]
    assert execution_order == expected, f"Expected {expected}, got {execution_order}"
    assert len(chats) == 10


# ===== Tests for parallel_chat_structured() =====


@pytest.mark.asyncio
async def test_parallel_chat_structured_basic():
    """Test parallel_chat_structured() without tools."""

    class Capital(BaseModel):
        """A capital city."""

        country: str
        capital: str

    chat = ChatOpenAI()

    countries = ["Canada", "Japan", "Brazil"]
    prompts = [
        f"What is the capital of {country}? Extract the country and capital."
        for country in countries
    ]

    results = await parallel_chat_structured(chat, prompts, Capital)

    assert len(results) == 3
    canada = results[0]
    japan = results[1]
    brazil = results[2]
    assert isinstance(canada.data, Capital)
    assert isinstance(japan.data, Capital)
    assert isinstance(brazil.data, Capital)
    # Verify we got reasonable results
    assert any("Ottawa" in r.data.capital for r in results)
    assert any("Tokyo" in r.data.capital for r in results)
    assert any("Brasilia" in r.data.capital or "Brasília" in r.data.capital for r in results)


@pytest.mark.asyncio
async def test_parallel_chat_structured_with_tools():
    """Test that parallel_chat_structured() works with tools."""
    call_log = []

    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        call_log.append(f"weather:{location}")
        return f"Weather in {location}: Sunny, 72°F"

    class WeatherReport(BaseModel):
        """A weather report."""

        location: str
        temperature: str
        conditions: str

    chat = ChatOpenAI(system_prompt="Be terse.")
    chat.register_tool(get_weather)

    prompts = [
        "Get the weather in Seattle and format it as a structured weather report",
        "Get the weather in Tokyo and format it as a structured weather report",
    ]

    results = await parallel_chat_structured(chat, prompts, WeatherReport)

    # Tools should have been called in order
    assert call_log == ["weather:Seattle", "weather:Tokyo"], f"Got {call_log}"

    # Results should be structured
    assert len(results) == 2
    seattle = results[0]
    tokyo = results[1]
    assert isinstance(seattle.data, WeatherReport)
    assert isinstance(tokyo.data, WeatherReport)
    assert seattle.data.location == "Seattle"
    assert tokyo.data.location == "Tokyo"


@pytest.mark.asyncio
async def test_parallel_chat_structured_tool_ordering():
    """Test that tool execution order is preserved in parallel_chat_structured()."""
    execution_order = []

    def fetch_data(id: str) -> str:
        """Fetch data by ID."""
        execution_order.append(id)
        return f"Data for {id}"

    class DataResult(BaseModel):
        """A data result."""

        id: str
        data: str

    chat = ChatOpenAI(system_prompt="Be terse. Use the fetch_data tool as requested.")
    chat.register_tool(fetch_data)

    prompts = [
        "Fetch data for ID 'A' and structure the result",
        "Fetch data for ID 'B' and structure the result",
        "Fetch data for ID 'C' and structure the result",
    ]

    results = await parallel_chat_structured(chat, prompts, DataResult)

    # Verify strict ordering
    assert execution_order == ["A", "B", "C"], f"Expected ['A', 'B', 'C'], got {execution_order}"
    assert len(results) == 3
    assert all(isinstance(r.data, DataResult) for r in results)
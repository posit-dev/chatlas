"""Test that parallel_chat() maintains tool execution order."""

import pytest
from chatlas import ChatOpenAI, parallel_chat


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_tool_ordering_basic():
    """Test that tools execute in prompt order across parallel chats."""
    # Create a tool that records execution order
    execution_order = []

    def record_tool(prompt_id: str) -> str:
        """Records which prompt's tool executed."""
        execution_order.append(prompt_id)
        return f"Executed for {prompt_id}"

    chat = ChatOpenAI()
    chat.register_tool(record_tool)

    prompts = [
        "Call record_tool with prompt_id='A'",
        "Call record_tool with prompt_id='B'",
        "Call record_tool with prompt_id='C'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Verify order - tools should execute in the order prompts were submitted
    assert execution_order == ["A", "B", "C"], (
        f"Expected ['A', 'B', 'C'], got {execution_order}"
    )

    # Verify all chats completed successfully
    assert len(chats) == 3
    for i, c in enumerate(chats):
        last_turn = c.get_last_turn()
        assert last_turn is not None
        # Each chat should have gotten a response
        assert last_turn.text is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_tool_ordering_multiple_tools_per_prompt():
    """Test that multiple tools within a prompt execute before the next prompt's tools."""
    execution_order = []

    def record_tool(prompt_id: str) -> str:
        """Records which prompt's tool executed."""
        execution_order.append(prompt_id)
        return f"Executed for {prompt_id}"

    chat = ChatOpenAI()
    chat.register_tool(record_tool)

    # Ask the model to call the tool multiple times per prompt
    prompts = [
        "Call record_tool twice: first with 'A1', then with 'A2'",
        "Call record_tool twice: first with 'B1', then with 'B2'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Should see all of A's tools before any of B's tools
    # The order should be ['A1', 'A2', 'B1', 'B2'], not interleaved like ['A1', 'B1', 'A2', 'B2']
    assert len(execution_order) == 4, (
        f"Expected 4 tool calls, got {len(execution_order)}"
    )

    # Find where A tools end and B tools begin
    a_tools = [x for x in execution_order if x.startswith("A")]
    b_tools = [x for x in execution_order if x.startswith("B")]

    # All A tools should come before all B tools
    last_a_index = max(execution_order.index(x) for x in a_tools)
    first_b_index = min(execution_order.index(x) for x in b_tools)

    assert last_a_index < first_b_index, (
        f"Expected all A tools before B tools, got {execution_order}"
    )


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_no_tools():
    """Test that parallel_chat works normally when no tools are needed."""
    chat = ChatOpenAI()

    prompts = [
        "Say 'Hello'",
        "Say 'World'",
        "Say 'Test'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Should complete successfully without any tool execution
    assert len(chats) == 3
    for c in chats:
        last_turn = c.get_last_turn()
        assert last_turn is not None
        assert last_turn.text is not None


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_parallel_chat_mixed_tools_and_no_tools():
    """Test parallel_chat with mix of prompts that do and don't use tools."""
    execution_order = []

    def record_tool(prompt_id: str) -> str:
        """Records which prompt's tool executed."""
        execution_order.append(prompt_id)
        return f"Executed for {prompt_id}"

    chat = ChatOpenAI()
    chat.register_tool(record_tool)

    prompts = [
        "Call record_tool with prompt_id='A'",
        "Just say 'Hello' without calling any tools",
        "Call record_tool with prompt_id='B'",
    ]

    chats = await parallel_chat(chat, prompts)

    # Only A and B should have called the tool, and in that order
    assert execution_order == ["A", "B"], f"Expected ['A', 'B'], got {execution_order}"

    assert len(chats) == 3

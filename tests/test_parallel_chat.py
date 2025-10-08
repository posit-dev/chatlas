"""Tests for parallel chat functionality."""

import pytest
from pydantic import BaseModel

from chatlas import (
    ChatOpenAI,
    ContentToolRequest,
    ContentToolResult,
    parallel_chat,
    parallel_chat_structured,
    parallel_chat_text,
)


@pytest.mark.asyncio
async def test_parallel_chat_basic():
    chat = ChatOpenAI(system_prompt="Be terse.")

    prompts = [
        "What is 1 + 1?",
        "What is 2 + 2?",
    ]

    chats = await parallel_chat(chat, prompts)

    assert len(chats) == 2
    assert chats[0] is not None
    turn1 = chats[0].get_last_turn()
    assert turn1 is not None
    assert "2" in turn1.text
    assert chats[1] is not None
    turn2 = chats[1].get_last_turn()
    assert turn2 is not None
    assert "4" in turn2.text

    chats = await parallel_chat_text(chat, prompts)
    assert len(chats) == 2
    assert chats[0] is not None
    assert "2" in chats[0]
    assert chats[1] is not None
    assert "4" in chats[1]


def new_roll_func():
    num_rolls = 0

    def roll():
        """Rolls a six-sided die"""
        nonlocal num_rolls
        num_rolls += 1
        return num_rolls

    return roll


@pytest.mark.asyncio
async def test_parallel_chat_tools():
    chat = ChatOpenAI(system_prompt="Be terse.")

    prompts = [
        "Roll the dice, please! Reply with 'You rolled ____'",
        "Roll the dice again! Reply with 'You rolled ____'",
    ]

    chat.register_tool(new_roll_func(), name="roll")
    chats = await parallel_chat(chat, prompts)

    assert len(chats) == 2
    assert chats[0] is not None
    turns = chats[0].get_turns()
    assert len(turns) == 4
    assert isinstance(turns[1].contents[0], ContentToolRequest)
    result = turns[2].contents[0]
    assert isinstance(result, ContentToolResult)
    assert result.name == "roll"
    assert f"You rolled {result.get_model_value()}" in turns[3].text

    turns = chats[0].get_turns()
    assert len(turns) == 4
    result = turns[2].contents[0]
    assert isinstance(result, ContentToolResult)
    assert result.name == "roll"
    assert f"You rolled {result.get_model_value()}" in turns[3].text


@pytest.mark.asyncio
async def test_parallel_chat_tools_uneven():
    chat = ChatOpenAI(system_prompt="Be terse.")

    prompts = [
        "Roll the dice, please! Reply with 'You rolled ____'",
        "reply with the word 'boop'",
        "Roll the dice again! Reply with 'You rolled ____'",
        "reply with the word 'beep'",
    ]

    chat.register_tool(new_roll_func(), name="roll")
    chats = await parallel_chat(chat, prompts)

    assert len(chats) == 4
    assert chats[0] is not None
    assert len(chats[0].get_turns()) == 4
    assert chats[1] is not None
    assert len(chats[1].get_turns()) == 2
    assert chats[2] is not None
    assert len(chats[2].get_turns()) == 4
    assert chats[3] is not None
    assert len(chats[3].get_turns()) == 2

    turn1 = chats[0].get_last_turn()
    assert turn1 is not None
    assert "You rolled 1" in turn1.text
    turn2 = chats[1].get_last_turn()
    assert turn2 is not None
    assert "boop" in turn2.text
    turn3 = chats[2].get_last_turn()
    assert turn3 is not None
    assert "You rolled 2" in turn3.text
    turn4 = chats[3].get_last_turn()
    assert turn4 is not None
    assert "beep" in turn4.text


@pytest.mark.asyncio
async def test_parallel_chat_structured():
    """Test parallel_chat_structured for structured data extraction."""
    chat = ChatOpenAI()

    prompts = ["John, age 15", "Jane, age 16"]

    class Person(BaseModel):
        """A person with name and age."""

        name: str
        age: int

    people = await parallel_chat_structured(chat, prompts, Person)

    assert len(people) == 2
    assert isinstance(people[0], Person)
    assert people[0].name.lower() == "john"
    assert people[0].age == 15
    assert isinstance(people[1], Person)
    assert people[1].name.lower() == "jane"
    assert people[1].age == 16

import re
import tempfile

import pytest
from pydantic import BaseModel

from chatlas import ChatOpenAI, ToolResult, Turn


def test_simple_batch_chat():
    chat = ChatOpenAI()
    response = chat.chat("What's 1 + 1. Just give me the answer, no punctuation")
    assert str(response) == "2"


@pytest.mark.asyncio
async def test_simple_async_batch_chat():
    chat = ChatOpenAI()
    response = await chat.chat_async(
        "What's 1 + 1. Just give me the answer, no punctuation",
    )
    assert "2" == await response.get_content()


def test_simple_streaming_chat():
    chat = ChatOpenAI()
    res = chat.stream("""
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """)
    chunks = [chunk for chunk in res]
    assert len(chunks) > 2
    result = "".join(chunks)
    rainbow_re = "^red *\norange *\nyellow *\ngreen *\nblue *\nindigo *\nviolet *\n?$"
    assert re.match(rainbow_re, result.lower())
    turn = chat.get_last_turn()
    assert turn is not None
    assert re.match(rainbow_re, turn.text.lower())


@pytest.mark.asyncio
async def test_simple_streaming_chat_async():
    chat = ChatOpenAI()
    res = await chat.stream_async("""
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """)
    chunks = [chunk async for chunk in res]
    assert len(chunks) > 2
    result = "".join(chunks)
    rainbow_re = "^red *\norange *\nyellow *\ngreen *\nblue *\nindigo *\nviolet *\n?$"
    assert re.match(rainbow_re, result.lower())
    turn = chat.get_last_turn()
    assert turn is not None
    assert re.match(rainbow_re, turn.text.lower())


def test_basic_repr(snapshot):
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output",
        turns=[
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ],
    )
    assert snapshot == repr(chat)


def test_basic_str(snapshot):
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output",
        turns=[
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ],
    )
    assert snapshot == str(chat)


def test_basic_export(snapshot):
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output",
        turns=[
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = tmpdir + "/chat.html"
        chat.export(tmpfile, title="My Chat")
        with open(tmpfile, "r") as f:
            assert snapshot == f.read()


def test_tool_results():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    def get_date():
        """Gets the current date"""
        return ToolResult("2024-01-01", response_output=["Tool result..."])

    chat.register_tool(get_date)
    chat.on_tool_request(lambda req: [f"Requesting tool {req.name}..."])

    results = []
    for chunk in chat.stream("What's the date?"):
        results.append(chunk)

    # Make sure values haven't been str()'d yet
    assert ["Requesting tool get_date..."] in results
    assert ["Tool result..."] in results

    response_str = "".join(str(chunk) for chunk in results)

    assert "Requesting tool get_date..." in response_str
    assert "Tool result..." in response_str
    assert "2024-01-01" in response_str

    chat.register_tool(get_date, on_request=lambda req: f"Calling {req.name}...")

    response = chat.chat("What's the date?")
    assert "Calling get_date..." in str(response)
    assert "Requesting tool get_date..." not in str(response)
    assert "Tool result..." in str(response)
    assert "2024-01-01" in str(response)


@pytest.mark.asyncio
async def test_tool_results_async():
    chat = ChatOpenAI(system_prompt="Be very terse, not even punctuation.")

    async def get_date():
        """Gets the current date"""
        import asyncio

        await asyncio.sleep(0.1)
        return ToolResult("2024-01-01", response_output=["Tool result..."])

    chat.register_tool(get_date)
    chat.on_tool_request(lambda req: [f"Requesting tool {req.name}..."])

    results = []
    async for chunk in await chat.stream_async("What's the date?"):
        results.append(chunk)

    # Make sure values haven't been str()'d yet
    assert ["Requesting tool get_date..."] in results
    assert ["Tool result..."] in results

    response_str = "".join(str(chunk) for chunk in results)

    assert "Requesting tool get_date..." in response_str
    assert "Tool result..." in response_str
    assert "2024-01-01" in response_str

    chat.register_tool(get_date, on_request=lambda req: [f"Calling {req.name}..."])

    response = await chat.chat_async("What's the date?")
    assert "Calling get_date..." in await response.get_content()
    assert "Requesting tool get_date..." not in await response.get_content()
    assert "Tool result..." in await response.get_content()
    assert "2024-01-01" in await response.get_content()


def test_extract_data():
    chat = ChatOpenAI()

    class Person(BaseModel):
        name: str
        age: int

    data = chat.extract_data("John, age 15, won first prize", data_model=Person)
    assert data == dict(name="John", age=15)


@pytest.mark.asyncio
async def test_extract_data_async():
    chat = ChatOpenAI()

    class Person(BaseModel):
        name: str
        age: int

    data = await chat.extract_data_async(
        "John, age 15, won first prize", data_model=Person
    )
    assert data == dict(name="John", age=15)


def test_last_turn_retrieval():
    chat = ChatOpenAI()
    assert chat.get_last_turn(role="user") is None
    assert chat.get_last_turn(role="assistant") is None

    chat.chat("Hi")
    user_turn = chat.get_last_turn(role="user")
    assert user_turn is not None and user_turn.role == "user"
    turn = chat.get_last_turn(role="assistant")
    assert turn is not None and turn.role == "assistant"


def test_system_prompt_retrieval():
    chat1 = ChatOpenAI()
    assert chat1.system_prompt is None
    assert chat1.get_last_turn(role="system") is None

    chat2 = ChatOpenAI(system_prompt="You are from New Zealand")
    assert chat2.system_prompt == "You are from New Zealand"
    turn = chat2.get_last_turn(role="system")
    assert turn is not None and turn.text == "You are from New Zealand"


def test_modify_system_prompt():
    chat = ChatOpenAI(
        turns=[
            Turn("user", "Hi"),
            Turn("assistant", "Hello"),
        ]
    )

    # NULL -> NULL
    chat.system_prompt = None
    assert chat.system_prompt is None

    # NULL -> string
    chat.system_prompt = "x"
    assert chat.system_prompt == "x"

    # string -> string
    chat.system_prompt = "y"
    assert chat.system_prompt == "y"

    # string -> NULL
    chat.system_prompt = None
    assert chat.system_prompt is None

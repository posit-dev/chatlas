import re
import tempfile

import pytest
from chatlas import (
    ChatOpenAI,
    ContentToolRequest,
    ContentToolResult,
    ToolRejectError,
    Turn,
)
from chatlas._chat import ToolFailureWarning
from pydantic import BaseModel


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
    res = chat.stream(
        """
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """
    )
    chunks = [chunk for chunk in res]
    assert len(chunks) > 2
    result = "".join(chunks)
    res = re.sub(r"\s+", "", result).lower()
    assert res == "redorangeyellowgreenblueindigoviolet"
    turn = chat.get_last_turn()
    assert turn is not None
    res = re.sub(r"\s+", "", turn.text).lower()
    assert res == "redorangeyellowgreenblueindigoviolet"


@pytest.mark.asyncio
async def test_simple_streaming_chat_async():
    chat = ChatOpenAI()
    res = await chat.stream_async(
        """
        What are the canonical colors of the ROYGBIV rainbow?
        Put each colour on its own line. Don't use punctuation.
    """
    )
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
        system_prompt="You're a helpful assistant that returns very minimal output"
    )
    chat.set_turns(
        [
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ]
    )
    assert snapshot == repr(chat)


def test_basic_str(snapshot):
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output"
    )
    chat.set_turns(
        [
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ]
    )
    assert snapshot == str(chat)


def test_basic_export(snapshot):
    chat = ChatOpenAI(
        system_prompt="You're a helpful assistant that returns very minimal output"
    )
    chat.set_turns(
        [
            Turn("user", "What's 1 + 1? What's 1 + 2?"),
            Turn("assistant", "2  3", tokens=(15, 5)),
        ]
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfile = tmpdir + "/chat.html"
        chat.export(tmpfile, title="My Chat")
        with open(tmpfile, "r") as f:
            assert snapshot == f.read()


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
    chat = ChatOpenAI()
    chat.set_turns(
        [
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


def test_json_serialize():
    chat = ChatOpenAI()
    chat.chat("Tell me a short joke", echo="none")
    turns = chat.get_turns()
    turns_json = [x.model_dump_json() for x in turns]
    turns_restored = [Turn.model_validate_json(x) for x in turns_json]
    assert len(turns) == 2
    # Completion objects, at least of right now, aren't included in the JSON
    turns[1].completion = None
    assert turns == turns_restored


# Chat can be deepcopied/forked
def test_deepcopy_chat():
    import copy

    chat = ChatOpenAI()
    chat.chat("Hi", echo="none")
    chat_fork = copy.deepcopy(chat)

    assert len(chat.get_turns()) == 2
    assert len(chat_fork.get_turns()) == 2

    chat_fork.chat("Bye", echo="none")

    assert len(chat.get_turns()) == 2
    assert len(chat_fork.get_turns()) == 4


def test_chat_callbacks():
    chat = ChatOpenAI()

    def test_tool(user: str) -> str:
        "Find out a user's favorite color"
        return "red"

    chat.register_tool(test_tool)

    last_request = None
    cb_count_request = 0
    cb_count_result = 0

    def on_tool_request(request: ContentToolRequest):
        nonlocal cb_count_request, last_request
        cb_count_request += 1
        assert isinstance(request, ContentToolRequest)
        assert request.name == "test_tool"
        last_request = request

    def on_tool_result(result: ContentToolResult):
        nonlocal cb_count_result, last_request
        cb_count_result += 1
        assert isinstance(result, ContentToolResult)
        assert result.request == last_request

    chat.on_tool_request(on_tool_request)
    chat.on_tool_result(on_tool_result)
    chat.chat("What are Joe and Hadley's favorite colors?")

    assert cb_count_request == 2
    assert cb_count_result == 2


@pytest.mark.filterwarnings("ignore", category=ToolFailureWarning)
def test_chat_tool_request_reject():
    chat = ChatOpenAI()

    def test_tool(user: str) -> str:
        "Find out a user's favorite color"
        return "red"

    chat.register_tool(test_tool)

    def on_tool_request(request: ContentToolRequest):
        if request.arguments["user"] == "Joe":
            raise ToolRejectError("Joe denied the request.")

    chat.on_tool_request(on_tool_request)

    response = chat.chat(
        "What are Joe and Hadley's favorite colors? ",
        "Write 'Joe ____ Hadley ____'. Use 'unknown' if you don't know. ",
        "Don't ever include punctuation in your answers.",
    )

    assert str(response).lower() == "joe unknown hadley red"


@pytest.mark.filterwarnings("ignore", category=ToolFailureWarning)
def test_chat_tool_request_reject2(capsys):
    chat = ChatOpenAI()

    def test_tool(user: str) -> str:
        "Find out a user's favorite color"
        if "joe" in user.lower():
            raise ToolRejectError("Joe denied the request.")
        return "red"

    chat.register_tool(test_tool)

    response = chat.chat(
        "What are Joe and Hadley's favorite colors? ",
        "Write 'Joe ____ Hadley ____'. Use 'unknown' if you don't know. ",
        "Don't ever include punctuation in your answers.",
    )

    assert str(response).lower() == "joe unknown hadley red"
    assert "Joe denied the request." in capsys.readouterr().out


def test_get_cost():
    chat = ChatOpenAI(api_key="fake_key")
    chat.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10)),
        ]
    )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Expected `options` to be one of 'all' or 'last', not 'bad_option'"
        ),
    ):
        chat.get_cost(options="bad_option")  # type: ignore

    # Checking that these have the right form vs. the actual calculation because the price may change
    cost = chat.get_cost(options="all")
    assert isinstance(cost, float)
    assert cost > 0

    last = chat.get_cost(options="last")
    assert isinstance(last, float)
    assert last > 0

    assert cost > last

    byoc = (2.0, 3.0)

    cost2 = chat.get_cost(options="all", token_price=byoc)
    assert cost2 == 0.000092

    last2 = chat.get_cost(options="last", token_price=byoc)
    assert last2 == 0.00003

    chat2 = ChatOpenAI(api_key="fake_key", model="BADBAD")
    chat2.set_turns(
        [
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(2, 10)),
            Turn(role="user", contents="Hi"),
            Turn(role="assistant", contents="Hello", tokens=(14, 10)),
        ]
    )
    with pytest.raises(
        KeyError,
        match="We could not locate pricing information for model 'BADBAD' from provider 'OpenAI'. If you know the pricing for this model, specify it in `token_price`.",
    ):
        chat2.get_cost(options="all")

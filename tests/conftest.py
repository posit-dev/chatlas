import tempfile
from pathlib import Path
from typing import Callable

import pytest
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, wait_exponential

from chatlas import (
    Chat,
    ContentToolRequest,
    ContentToolResult,
    Turn,
    content_image_file,
    content_image_url,
    content_pdf_file,
)

ChatFun = Callable[..., Chat]


class ArticleSummary(BaseModel):
    """Summary of the article"""

    title: str
    author: str


article = """
# Apples are tasty

By Hadley Wickham
Apples are delicious and tasty and I like to eat them.
Except for red delicious, that is. They are NOT delicious.
"""


def assert_turns_system(chat_fun: ChatFun):
    system_prompt = "Return very minimal output, AND ONLY USE UPPERCASE."

    chat = chat_fun(system_prompt=system_prompt)
    response = chat.chat("What is the name of Winnie the Pooh's human friend?")
    response_text = str(response)
    assert len(chat.get_turns()) == 2
    assert "CHRISTOPHER ROBIN" in response_text.upper()

    chat = chat_fun()
    chat.system_prompt = system_prompt
    response = chat.chat("What is the name of Winnie the Pooh's human friend?")
    assert "CHRISTOPHER ROBIN" in str(response).upper()
    assert len(chat.get_turns()) == 2


def assert_turns_existing(chat_fun: ChatFun):
    chat = chat_fun(system_prompt="Return very minimal output; no punctuation.")
    chat.set_turns(
        [
            Turn("user", "List the names of any 8 of Santa's 9 reindeer."),
            Turn(
                "assistant",
                "Dasher, Dancer, Vixen, Comet, Cupid, Donner, Blitzen, and Rudolph.",
            ),
        ]
    )

    assert len(chat.get_turns()) == 2

    response = chat.chat("Who is the remaining one? Just give the name")
    assert "Prancer" in str(response)
    assert len(chat.get_turns()) == 4


def assert_tools_simple(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(
        system_prompt="Always use a tool to help you answer. Reply with 'It is ____.'."
    )

    def get_date():
        """Gets the current date"""
        return "2024-01-01"

    chat.register_tool(get_date)

    response = chat.chat("What's the current date in Y-M-D format?", stream=stream)
    assert "2024-01-01" in str(response)

    response = chat.chat("What month is it? Provide the full name.", stream=stream)
    assert "January" in str(response)


def assert_tools_simple_stream_content(chat_fun: ChatFun):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    def get_date():
        """Gets the current date"""
        return "2024-01-01"

    chat.register_tool(get_date)

    response = chat.stream("What's the current date in Y-M-D format?", content="all")
    chunks = [chunk for chunk in response]
    request = [x for x in chunks if isinstance(x, ContentToolRequest)]
    assert len(request) == 1
    response = [x for x in chunks if isinstance(x, ContentToolResult)]
    assert len(response) == 1
    str_response = "".join([str(x) for x in chunks])
    assert "2024-01-01" in str_response
    assert "get_date" in str_response


async def assert_tools_async(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    async def get_current_date():
        """Gets the current date"""
        import asyncio

        await asyncio.sleep(0.1)
        return "2024-01-01"

    chat.register_tool(get_current_date)

    response = await chat.chat_async(
        "What's the current date in Y-M-D format?", stream=stream
    )
    assert "2024-01-01" in await response.get_content()

    # Can't use async tools in a synchronous chat...
    with pytest.raises(Exception, match="async tools in a synchronous chat"):
        str(chat.chat("Great. Do it again.", stream=stream))

    # ... but we can use synchronous tools in an async chat
    def get_current_date2():
        """Gets the current date"""
        return "2024-01-01"

    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")
    chat.register_tool(get_current_date2)

    response = await chat.chat_async(
        "What's the current date in Y-M-D format?", stream=stream
    )
    assert "2024-01-01" in await response.get_content()


def assert_tools_parallel(
    chat_fun: ChatFun, *, total_calls: int = 4, stream: bool = True
):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    def favorite_color(person: str):
        """Returns a person's favourite colour"""
        return "sage green" if person == "Joe" else "red"

    chat.register_tool(favorite_color)

    response = chat.chat(
        """
        What are Joe and Hadley's favourite colours?
        Answer like name1: colour1, name2: colour2
    """,
        stream=stream,
    )

    res = str(response).replace(":", "")
    assert "Joe sage green" in res
    assert "Hadley red" in res
    assert len(chat.get_turns()) == total_calls


def assert_tools_sequential(chat_fun: ChatFun, total_calls: int, stream: bool = True):
    chat = chat_fun(
        system_prompt="""
        Be very terse, not even punctuation. If asked for equipment to pack,
        first use the weather_forecast tool provided to you. Then, use the
        equipment tool provided to you.
        """
    )

    def weather_forecast(city: str):
        """Gets the weather forecast for a city"""
        return "rainy" if city == "New York" else "sunny"

    chat.register_tool(weather_forecast)

    def equipment(weather: str):
        """Gets the equipment needed for a weather condition"""
        return "umbrella" if weather == "rainy" else "sunscreen"

    chat.register_tool(equipment)

    response = chat.chat(
        "What should I pack for New York this weekend?",
        stream=stream,
    )
    assert "umbrella" in str(response).lower()
    assert len(chat.get_turns()) == total_calls


def assert_data_extraction(chat_fun: ChatFun):
    chat = chat_fun()
    data = chat.extract_data(article, data_model=ArticleSummary)
    assert isinstance(data, dict)
    assert data["author"] == "Hadley Wickham"
    assert data["title"].lower() == "apples are tasty"
    data2 = chat.extract_data(article, data_model=ArticleSummary)
    assert data2["author"] == "Hadley Wickham"
    assert data2["title"].lower() == "apples are tasty"


def assert_images_inline(chat_fun: ChatFun, stream: bool = True):
    img = Image.new("RGB", (60, 30), color="red")
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test_image.png"
        img.save(img_path)
        chat = chat_fun()
        response = chat.chat(
            "What's in this image?",
            content_image_file(str(img_path), resize="low"),
            stream=stream,
        )
        assert "red" in str(response).lower()


def assert_images_remote(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun()
    response = chat.chat(
        "What's in this image? (Be sure to mention the outside shape)",
        content_image_url("https://httr2.r-lib.org/logo.png"),
        stream=stream,
    )
    assert "hex" in str(response).lower()
    assert "baseball" in str(response).lower()


def assert_images_remote_error(chat_fun: ChatFun):
    chat = chat_fun()
    image_remote = content_image_url("https://httr2.r-lib.org/logo.png")

    with pytest.raises(Exception, match="Remote images aren't supported"):
        chat.chat("What's in this image?", image_remote)

    assert len(chat.get_turns()) == 0


def assert_pdf_local(chat_fun: ChatFun):
    chat = chat_fun()
    apples = Path(__file__).parent / "apples.pdf"
    response = chat.chat(
        "What's the title of this document?",
        content_pdf_file(apples),
    )
    assert "apples are tasty" in str(response).lower()

    response = chat.chat(
        "What apple is not tasty according to the document?",
        "Two word answer only.",
    )
    assert "red delicious" in str(response).lower()


retry_api_call = retry(
    wait=wait_exponential(min=1, max=60),
    reraise=True,
)


@pytest.fixture
def test_images_dir():
    return Path(__file__).parent / "images"

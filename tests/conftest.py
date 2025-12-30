import os
import tempfile
from pathlib import Path
from typing import Callable

import pytest
from chatlas import (
    AssistantTurn,
    Chat,
    ContentToolRequest,
    ContentToolResult,
    UserTurn,
    content_image_file,
    content_image_url,
    content_pdf_file,
)
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, wait_exponential

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
    chat = chat_fun()
    chat.set_turns(
        [
            UserTurn("My name is Steve"),
            AssistantTurn(
                "Hello Steve, how can I help you today?",
            ),
        ]
    )

    assert len(chat.get_turns()) == 2

    response = chat.chat("What is my name?")
    assert "steve" in str(response).lower()
    assert len(chat.get_turns()) == 4


def assert_tools_simple(chat_fun: ChatFun, stream: bool = True):
    chat = chat_fun(
        system_prompt="Always use a tool to help you answer. Reply with 'It is ____.'."
    )

    def get_date():
        """Gets the current date"""
        return "2024-01-01"

    chat.register_tool(get_date)

    response = chat.chat("What's the current date in YYYY-MM-DD format?", stream=stream)
    assert "2024-01-01" in str(response)

    response = chat.chat("What month is it? Provide the full name.", stream=stream)
    assert "January" in str(response)


def assert_tools_simple_stream_content(chat_fun: ChatFun):
    from chatlas._content import ToolAnnotations

    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    def get_date():
        """Gets the current date"""
        return "2024-01-01"

    chat.register_tool(get_date, annotations=ToolAnnotations(title="Get Date"))

    response = chat.stream(
        "What's the current date in YYYY-MM-DD format?", content="all"
    )
    chunks = [chunk for chunk in response]

    # Emits a request with tool annotations
    request = [x for x in chunks if isinstance(x, ContentToolRequest)]
    assert len(request) == 1
    assert request[0].name == "get_date"
    assert request[0].tool is not None
    assert request[0].tool.name == "get_date"
    assert request[0].tool.annotations is not None
    assert request[0].tool.annotations["title"] == "Get Date"

    # Emits a response (with a reference to the request)
    response = [x for x in chunks if isinstance(x, ContentToolResult)]
    assert len(response) == 1
    assert response[0].request == request[0]

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
        "What's the current date in YYYY-MM-DD format?", stream=stream
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
        "What's the current date in YYYY-MM-DD format?", stream=stream
    )
    assert "2024-01-01" in await response.get_content()


def assert_tools_parallel(
    chat_fun: ChatFun, *, total_calls: int = 4, stream: bool = True
):
    chat = chat_fun(system_prompt="Be very terse, not even punctuation.")

    def favorite_color(_person: str):
        """Returns a person's favourite colour"""
        return "sage green" if _person == "Joe" else "red"

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
    data = chat.chat_structured(article, data_model=ArticleSummary)
    assert isinstance(data, ArticleSummary)
    assert data.author == "Hadley Wickham"
    assert data.title.lower() == "apples are tasty"
    data2 = chat.chat_structured(article, data_model=ArticleSummary)
    assert data2.author == "Hadley Wickham"
    assert data2.title.lower() == "apples are tasty"

    class Person(BaseModel):
        name: str
        age: int

    data = chat.chat_structured(
        "Generate the name and age of a random person.", data_model=Person
    )
    response = chat.chat("What is the name of the person?")
    assert data.name in str(response)


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


def assert_list_models(chat_fun: ChatFun):
    chat = chat_fun()
    models = chat.list_models()
    assert models is not None
    assert isinstance(models, list)
    assert len(models) > 0, (
        f"{chat_fun.__name__}().list_models() returned an empty list"
    )
    assert "id" in models[0]


retry_api_call = retry(
    wait=wait_exponential(min=1, max=60),
    reraise=True,
)


@pytest.fixture
def test_images_dir():
    return Path(__file__).parent / "images"


@pytest.fixture
def test_batch_dir():
    return Path(__file__).parent / "batch"


# ---------------------------------------------------------------------------
# VCR Configuration for HTTP recording/replay
# ---------------------------------------------------------------------------


# Some providers (Anthropic, Google) validate API keys before making requests,
# so we need dummy keys for VCR replay mode
@pytest.fixture(autouse=True, scope="session")
def set_dummy_api_keys_for_vcr():
    """Set dummy API keys for providers that validate keys before requests.

    This enables VCR replay mode to work without real API keys.
    When a real key is set, it takes precedence.
    """
    dummy_keys = {
        "OPENAI_API_KEY": "sk-dummy-openai-key-for-vcr-replay",
        "ANTHROPIC_API_KEY": "sk-ant-dummy-key-for-vcr-replay",
        "GOOGLE_API_KEY": "dummy-google-key-for-vcr-replay",
        "AZURE_OPENAI_API_KEY": "dummy-azure-key-for-vcr-replay",
        "CLOUDFLARE_API_KEY": "dummy-cloudflare-key-for-vcr-replay",
        "CLOUDFLARE_ACCOUNT_ID": "dummy-cloudflare-account-for-vcr-replay",
        "DATABRICKS_HOST": "https://dummy-databricks-host.cloud.databricks.com",
        "DATABRICKS_TOKEN": "dummy-databricks-token-for-vcr-replay",
        "DEEPSEEK_API_KEY": "dummy-deepseek-key-for-vcr-replay",
        "GITHUB_TOKEN": "dummy-github-token-for-vcr-replay",
        "HUGGINGFACE_API_KEY": "dummy-huggingface-key-for-vcr-replay",
        "MISTRAL_API_KEY": "dummy-mistral-key-for-vcr-replay",
        "OPENROUTER_API_KEY": "dummy-openrouter-key-for-vcr-replay",
    }
    original_values = {}

    for key, dummy_value in dummy_keys.items():
        original_values[key] = os.environ.get(key)
        if not original_values[key]:
            os.environ[key] = dummy_value

    yield

    # Restore original values
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


def _filter_response_headers(response):
    """Remove sensitive headers from response before recording."""
    headers_to_remove = [
        "openai-organization",
        "openai-project",
        "anthropic-organization-id",
        "set-cookie",
        "cf-ray",
        "x-request-id",
        "request-id",
    ]
    headers = response.get("headers", {})
    for header in headers_to_remove:
        # Headers can be stored as lowercase or original case
        headers.pop(header, None)
        headers.pop(header.title(), None)
        headers.pop(header.lower(), None)
        # Also handle capitalization variations
        headers.pop(header.replace("-", "").lower(), None)
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Global VCR configuration for pytest-recording."""
    return {
        "filter_headers": [
            "authorization",
            "x-api-key",
            "api-key",
            "openai-organization",
            "x-goog-api-key",
            "x-stainless-arch",
            "x-stainless-lang",
            "x-stainless-os",
            "x-stainless-package-version",
            "x-stainless-runtime",
            "x-stainless-runtime-version",
            "x-stainless-retry-count",
            "user-agent",
        ],
        "filter_post_data_parameters": ["api_key"],
        "decode_compressed_response": True,
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
        "before_record_response": _filter_response_headers,
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Store cassettes in per-module directories."""
    module_name = request.module.__name__.split(".")[-1]
    return os.path.join(os.path.dirname(__file__), "_vcr", module_name)


def pytest_exception_interact(node, call, report):
    """Provide helpful message when VCR cassette is missing."""
    if call.excinfo is not None:
        exc_str = str(call.excinfo.value)
        if "CannotOverwriteExistingCassetteException" in exc_str or "Can't find" in exc_str:
            print("\n" + "=" * 60)
            print("VCR CASSETTE MISSING OR OUTDATED")
            print("=" * 60)
            print("To record/update cassettes, run locally with API keys:")
            print("  make record-vcr")
            print("Or for a specific provider:")
            print("  make record-vcr-openai")
            print("  make record-vcr-anthropic")
            print("  make record-vcr-google")
            print("")
            print("Or directly with pytest:")
            print("  uv run pytest --record-mode=all tests/test_provider_openai.py -v")
            print("=" * 60 + "\n")

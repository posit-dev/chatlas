import os
from pathlib import Path
from typing import Callable

import pytest

# ---------------------------------------------------------------------------
# Dummy API keys for VCR replay testing
# ---------------------------------------------------------------------------
# These are set if not already present, allowing VCR tests to run without
# real credentials. When recording cassettes, set real API keys in your env.

_DUMMY_CREDENTIALS = {
    "ANTHROPIC_API_KEY": "dummy-anthropic-key",
    "AZURE_OPENAI_API_KEY": "dummy-azure-key",
    "CLOUDFLARE_API_KEY": "dummy-cloudflare-key",
    "CLOUDFLARE_ACCOUNT_ID": "dummy-cloudflare-id",
    "DATABRICKS_HOST": "dummy-databricks-host",
    "DATABRICKS_TOKEN": "dummy-databricks-token",
    "DEEPSEEK_API_KEY": "dummy-deepseek-key",
    "GH_TOKEN": "dummy-github-token",
    "GOOGLE_API_KEY": "dummy-google-key",
    "GROQ_API_KEY": "dummy-groq-key",
    "HUGGINGFACE_API_KEY": "dummy-huggingface-key",
    "MISTRAL_API_KEY": "dummy-mistral-key",
    "OPENAI_API_KEY": "dummy-openai-key",
    "OPENROUTER_API_KEY": "dummy-openrouter-key",
}


def pytest_configure(config):
    """Set dummy API keys for VCR replay if not already set."""
    for key, value in _DUMMY_CREDENTIALS.items():
        if key not in os.environ:
            os.environ[key] = value


def is_dummy_credential(env_var: str) -> bool:
    """
    Check if an environment variable contains a dummy credential.

    Use this to skip tests that require live API calls (e.g., multi-sample tests
    where VCR response ordering is unreliable).

    Example:
        @pytest.mark.skipif(
            is_dummy_credential("ANTHROPIC_API_KEY"),
            reason="This test requires live API calls",
        )
        def test_something():
            ...
    """
    value = os.environ.get(env_var, "")
    dummy_value = _DUMMY_CREDENTIALS.get(env_var, "")
    return value == dummy_value or value.startswith("dummy")


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
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

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
    # Use a fixture image with resize="none" to ensure deterministic VCR cassette
    # matching (resize can produce different bytes across platforms/PIL versions)
    img_path = Path(__file__).parent / "images" / "red_test.png"
    chat = chat_fun()
    response = chat.chat(
        "What's in this image?",
        content_image_file(str(img_path), resize="none"),
        stream=stream,
    )
    assert "red" in str(response).lower()


def assert_images_remote(
    chat_fun: ChatFun, stream: bool = True, test_shape: bool = True
):
    chat = chat_fun()
    response = chat.chat(
        "What's in this image? (Be sure to mention the outside shape)",
        content_image_url("https://httr2.r-lib.org/logo.png"),
        stream=stream,
    )
    assert "baseball" in str(response).lower()
    if test_shape:
        assert "hex" in str(response).lower()


def assert_images_remote_error(
    chat_fun: ChatFun, message: str = "Remote images aren't supported"
):
    chat = chat_fun()
    image_remote = content_image_url("https://httr2.r-lib.org/logo.png")

    with pytest.raises(Exception, match=message):
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
    stop=stop_after_attempt(3),
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


# Default matchers for VCR - most tests should match on body
VCR_MATCH_ON_DEFAULT = ["method", "scheme", "host", "port", "path", "body"]
# Some tests have dynamic request bodies (temp filenames, dynamic IDs) - skip body matching
VCR_MATCH_ON_WITHOUT_BODY = ["method", "scheme", "host", "port", "path"]


def make_vcr_config(match_on: list[str] = VCR_MATCH_ON_DEFAULT) -> dict:
    """
    Create a VCR configuration dictionary.

    Args:
        match_on: List of request attributes to match on. Use VCR_MATCH_ON_DEFAULT
                  for most tests, or VCR_MATCH_ON_WITHOUT_BODY for tests with
                  dynamic request bodies (e.g., temp filenames, generated IDs).

    Returns:
        VCR configuration dictionary suitable for pytest-recording.
    """
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
            # AWS Bedrock headers (kept here for convenience, harmless for non-AWS tests)
            "x-amz-sso_bearer_token",
            "X-Amz-Security-Token",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
        ],
        "filter_post_data_parameters": ["api_key"],
        "decode_compressed_response": True,
        "match_on": match_on,
        "before_record_response": _filter_response_headers,
    }


@pytest.fixture(scope="module")
def vcr_config():
    """Global VCR configuration for pytest-recording."""
    return make_vcr_config()


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    """Store cassettes in per-module directories."""
    module_name = request.module.__name__.split(".")[-1]
    return os.path.join(os.path.dirname(__file__), "_vcr", module_name)


def pytest_exception_interact(node, call, report):
    """Provide helpful message when VCR cassette is missing."""
    if call.excinfo is not None:
        exc_str = str(call.excinfo.value)
        if (
            "CannotOverwriteExistingCassetteException" in exc_str
            or "Can't find" in exc_str
        ):
            print("\n" + "=" * 60)
            print("VCR CASSETTE MISSING OR OUTDATED")
            print("=" * 60)
            print("To record/update all cassettes, run locally with API keys:")
            print("  make update-snaps-vcr")
            print("")
            print("Or record a specific test file:")
            print("  uv run pytest tests/test_provider_openai.py -v --record-mode=rewrite")
            print("")
            print("Or record a single test:")
            print("  uv run pytest tests/test_provider_openai.py::test_openai_simple_request -v --record-mode=rewrite")
            print("=" * 60 + "\n")

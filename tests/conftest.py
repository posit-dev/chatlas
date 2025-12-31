import os
import re
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


def _scrub_aws_credentials(response):
    """Scrub AWS credentials from response body before recording."""
    import json

    body = response.get("body", {})
    if not body:
        return response

    body_string = body.get("string", "")
    if not body_string:
        return response

    # Handle both string and bytes
    if isinstance(body_string, bytes):
        try:
            body_string = body_string.decode("utf-8")
            was_bytes = True
        except UnicodeDecodeError:
            return response
    else:
        was_bytes = False

    # Try to parse as JSON and scrub credentials
    try:
        data = json.loads(body_string)
        if "roleCredentials" in data:
            creds = data["roleCredentials"]
            if "accessKeyId" in creds:
                creds["accessKeyId"] = "SCRUBBED_ACCESS_KEY_ID"
            if "secretAccessKey" in creds:
                creds["secretAccessKey"] = "SCRUBBED_SECRET_ACCESS_KEY"
            if "sessionToken" in creds:
                creds["sessionToken"] = "SCRUBBED_SESSION_TOKEN"
            body_string = json.dumps(data)
    except (json.JSONDecodeError, TypeError):
        # Not JSON, try regex patterns for AWS credentials
        # Access key pattern: AKIA... or ASIA... (20 chars)
        body_string = re.sub(
            r"(A[SK]IA[A-Z0-9]{16})",
            "SCRUBBED_ACCESS_KEY_ID",
            body_string,
        )
        # Secret key pattern (40 chars of base64-ish characters after common prefixes)
        body_string = re.sub(
            r'("secretAccessKey"\s*:\s*")[^"]+(")',
            r"\1SCRUBBED_SECRET_ACCESS_KEY\2",
            body_string,
        )
        # Session token pattern
        body_string = re.sub(
            r'("sessionToken"\s*:\s*")[^"]+(")',
            r"\1SCRUBBED_SESSION_TOKEN\2",
            body_string,
        )

    if was_bytes:
        body["string"] = body_string.encode("utf-8")
    else:
        body["string"] = body_string

    return response


def _filter_aws_response(response):
    """Combined filter for response headers and AWS credentials."""
    response = _filter_response_headers(response)
    response = _scrub_aws_credentials(response)
    return response


def _scrub_aws_request(request):
    """Scrub AWS account ID from request URLs."""
    # Replace account_id parameter in SSO URLs
    if "account_id=" in request.uri:
        request.uri = re.sub(
            r"account_id=\d+",
            "account_id=SCRUBBED_ACCOUNT_ID",
            request.uri,
        )
    return request


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
            # AWS Bedrock headers
            "x-amz-sso_bearer_token",
            "X-Amz-Security-Token",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
        ],
        "filter_post_data_parameters": ["api_key"],
        "decode_compressed_response": True,
        "match_on": ["method", "scheme", "host", "port", "path", "body"],
        "before_record_response": _filter_aws_response,
        "before_record_request": _scrub_aws_request,
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
        if (
            "CannotOverwriteExistingCassetteException" in exc_str
            or "Can't find" in exc_str
        ):
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

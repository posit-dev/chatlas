import pytest
from chatlas import ChatOpenAI
from chatlas._content import ContentToolResultImage, ContentToolResultResource
from chatlas._tools import Tool
from chatlas.types import ContentToolRequest, ContentToolResult
from pydantic import BaseModel, Field


class TestNewToolConstructor:
    """Test the new Tool constructor that takes schema parameters directly."""

    def test_tool_constructor_with_schema(self):
        """Test Tool constructor with explicit parameters."""

        def my_func(x: int, y: str) -> str:
            return f"{x}: {y}"

        parameters = {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "string"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        }

        tool = Tool(
            func=my_func,
            name="my_tool",
            description="A test tool",
            parameters=parameters,
        )

        assert tool.name == "my_tool"
        assert tool.func == my_func
        assert tool.schema["type"] == "function"
        func = tool.schema["function"]
        assert func["name"] == "my_tool"
        assert func.get("description") == "A test tool"
        assert func.get("parameters") == parameters

    def test_tool_constructor_async_function(self):
        """Test Tool constructor with async function."""

        async def async_func(x: int) -> int:
            return x * 2

        parameters = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
            "additionalProperties": False,
        }

        tool = Tool(
            func=async_func,
            name="async_tool",
            description="An async test tool",
            parameters=parameters,
        )

        assert tool.name == "async_tool"
        assert tool.func == async_func
        assert tool._is_async is True


class TestToolFromFunc:
    """Test Tool.from_func() class method."""

    def test_from_func_basic(self):
        """Test creating a Tool from a basic function."""

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        tool = Tool.from_func(add)

        assert tool.name == "add"
        assert tool.func == add

        func = tool.schema["function"]
        assert func["name"] == "add"
        assert func.get("description") == "Add two numbers."
        assert func.get("parameters") == {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "integer"},
            },
            "required": ["x", "y"],
        }

    def test_from_func_with_model(self):
        """Test creating a Tool from a function with custom Pydantic model."""

        class AddParams(BaseModel):
            """Parameters for adding numbers."""

            x: int = Field(description="First number")
            y: int = Field(description="Second number")

        def add(x: int, y: int) -> int:
            return x + y

        tool = Tool.from_func(add, model=AddParams)

        assert tool.name == "AddParams"
        assert tool.func == add

        func = tool.schema["function"]
        assert func["name"] == "AddParams"
        assert func.get("description") == "Parameters for adding numbers."

        # Check that Field descriptions are preserved
        params = func.get("parameters", {})
        props = params["properties"]
        assert props["x"]["description"] == "First number"  # type: ignore
        assert props["y"]["description"] == "Second number"  # type: ignore

    def test_from_func_model_mismatch_error(self):
        """Test that mismatched model fields and function parameters raise error."""

        class WrongParams(BaseModel):
            a: int
            b: int

        def add(x: int, y: int) -> int:
            return x + y

        with pytest.raises(ValueError, match="Fields found in one but not the other"):
            Tool.from_func(add, model=WrongParams)

    def test_from_func_no_docstring(self):
        """Test creating a Tool from a function without docstring."""

        def no_doc(x: int) -> int:
            return x

        tool = Tool.from_func(no_doc)

        assert tool.name == "no_doc"

        func = tool.schema["function"]
        assert func.get("description") == ""

    def test_from_func_async(self):
        """Test creating a Tool from async function."""

        async def async_add(x: int, y: int) -> int:
            """Add two numbers asynchronously."""
            return x + y

        tool = Tool.from_func(async_add)

        assert tool.name == "async_add"
        assert tool._is_async is True
        func = tool.schema["function"]
        assert func.get("description") == "Add two numbers asynchronously."


class TestNewContentClasses:
    """Test new ContentToolResultImage and ContentToolResultResource classes."""

    def test_content_tool_result_image(self):
        """Test ContentToolResultImage class."""
        import base64

        # Create dummy base64 image data
        image_data = base64.b64encode(b"fake image data").decode("utf-8")

        result = ContentToolResultImage(value=image_data, mime_type="image/png")

        assert result.content_type == "tool_result_image"
        assert result.value == image_data
        assert result.mime_type == "image/png"
        assert result.model_format == "as_is"
        assert "ContentToolResultImage" in str(result)
        assert "image/png" in str(result)

        # Test markdown representation
        markdown = result._repr_markdown_()
        assert f"![](data:image/png;base64,{image_data})" == markdown

    def test_content_tool_result_resource(self):
        """Test ContentToolResultResource class."""
        resource_data = b"This is some resource data"

        result = ContentToolResultResource(value=resource_data, mime_type="text/plain")

        assert result.content_type == "tool_result_resource"
        assert result.value == resource_data
        assert result.mime_type == "text/plain"
        assert result.model_format == "as_is"
        assert "ContentToolResultResource" in str(result)
        assert "text/plain" in str(result)

        # Test MIME bundle representation
        mime_bundle = result._repr_mimebundle_()
        assert mime_bundle["text/plain"] == "<text/plain object>"

    def test_content_tool_result_image_valid_mime_types(self):
        """Test that valid MIME types work correctly."""
        valid_types = ["image/png", "image/jpeg", "image/webp", "image/gif"]

        for mime_type in valid_types:
            result = ContentToolResultImage(value="base64data", mime_type=mime_type)
            assert result.mime_type == mime_type


class TestChatGetSetTools:
    """Test Chat.get_tools() and Chat.set_tools() methods."""

    def test_get_tools_empty(self):
        """Test get_tools() returns empty list initially."""
        chat = ChatOpenAI()
        tools = chat.get_tools()
        assert tools == []
        assert isinstance(tools, list)

    def test_get_tools_after_registration(self):
        """Test get_tools() returns registered tools."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        def subtract(x: int, y: int) -> int:
            return x - y

        chat.register_tool(add)
        chat.register_tool(subtract)

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"add", "subtract"}

    def test_set_tools_with_functions(self):
        """Test set_tools() with function list."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        def multiply(x: int, y: int) -> int:
            return x * y

        chat.set_tools([add, multiply])

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"add", "multiply"}

    def test_set_tools_with_tool_objects(self):
        """Test set_tools() with Tool objects."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        tool1 = Tool.from_func(add)
        tool2 = Tool(
            func=lambda x: x * 2,
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )

        chat.set_tools([tool1, tool2])

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"add", "double"}

    def test_set_tools_mixed(self):
        """Test set_tools() with mixed functions and Tool objects."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        tool = Tool(
            func=lambda x: x * 2,
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": False,
            },
        )

        chat.set_tools([add, tool])

        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"add", "double"}

    def test_set_tools_replaces_existing(self):
        """Test that set_tools() replaces existing tools."""
        chat = ChatOpenAI()

        def original(x: int) -> int:
            return x

        def new1(x: int) -> int:
            return x + 1

        def new2(x: int) -> int:
            return x + 2

        # Register original tool
        chat.register_tool(original)
        assert len(chat.get_tools()) == 1

        # Replace with new tools
        chat.set_tools([new1, new2])
        tools = chat.get_tools()
        assert len(tools) == 2
        tool_names = {tool.name for tool in tools}
        assert tool_names == {"new1", "new2"}


class TestRegisterToolForce:
    """Test register_tool() with force parameter and exception handling."""

    def test_register_tool_duplicate_name_error(self):
        """Test that registering duplicate tool name raises error."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            return x + y

        def another_add(a: int, b: int) -> int:
            return a + b

        # Rename second function to have same name as first
        another_add.__name__ = "add"

        chat.register_tool(add)

        with pytest.raises(
            ValueError, match="Tool with name 'add' is already registered"
        ):
            chat.register_tool(another_add)

    def test_register_tool_force_overwrite(self):
        """Test that force=True allows overwriting existing tool."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            """Original add function."""
            return x + y

        def new_add(x: int, y: int) -> int:
            """New add function."""
            return (x + y) * 2

        # Rename new function to have same name
        new_add.__name__ = "add"

        # Register original
        chat.register_tool(add)
        original_tool = chat._tools["add"]
        original_func = original_tool.schema["function"]
        assert original_func.get("description") == "Original add function."

        # Overwrite with force=True
        chat.register_tool(new_add, force=True)
        new_tool = chat._tools["add"]
        new_func = new_tool.schema["function"]
        assert new_func.get("description") == "New add function."
        assert new_tool.func == new_add
        assert len(chat._tools) == 1  # Should still be only one tool

    def test_register_tool_with_same_name_different_function(self):
        """Test registering functions with the same name but different implementations."""
        chat = ChatOpenAI()

        def my_tool(x: int) -> int:
            """First implementation."""
            return x

        # Change the function name to test explicit naming
        my_tool.__name__ = "test_tool"

        def another_func(x: int) -> int:
            """Second implementation."""
            return x * 2

        another_func.__name__ = "test_tool"

        chat.register_tool(my_tool)

        with pytest.raises(
            ValueError, match="Tool with name 'test_tool' is already registered"
        ):
            chat.register_tool(another_func)

        # But with force=True it should work
        chat.register_tool(another_func, force=True)
        assert chat._tools["test_tool"].func == another_func


class TestToolYielding:
    """Test tool functions that yield multiple results."""

    def test_tool_yielding_multiple_results(self):
        """Test tool function that yields multiple results."""
        chat = ChatOpenAI()

        def multi_result_tool(count: int):
            """Tool that yields multiple results."""
            for i in range(count):
                yield f"Result {i + 1}"

        chat.register_tool(multi_result_tool)

        request = ContentToolRequest(
            id="test-id", name="multi_result_tool", arguments={"count": 3}
        )

        results = list(chat._invoke_tool(request))

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, ContentToolResult)
            assert result.value == f"Result {i + 1}"
            assert result.request == request
            assert result.error is None

    def test_tool_yielding_single_result_still_works(self):
        """Test that regular (non-yielding) tools still work."""
        chat = ChatOpenAI()

        def single_result_tool(x: int) -> int:
            """Tool that returns single result."""
            return x * 2

        chat.register_tool(single_result_tool)

        request = ContentToolRequest(
            id="test-id", name="single_result_tool", arguments={"x": 5}
        )

        results = list(chat._invoke_tool(request))

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ContentToolResult)
        assert result.value == 10
        assert result.request == request
        assert result.error is None

    def test_tool_yielding_content_tool_results(self):
        """Test tool that yields ContentToolResult objects directly."""
        chat = ChatOpenAI()

        def custom_result_tool(count: int):
            """Tool that yields ContentToolResult objects."""
            for i in range(count):
                yield ContentToolResult(
                    value=f"Custom result {i + 1}", extra={"index": i}
                )

        chat.register_tool(custom_result_tool)

        request = ContentToolRequest(
            id="test-id", name="custom_result_tool", arguments={"count": 2}
        )

        results = list(chat._invoke_tool(request))

        assert len(results) == 2
        for i, result in enumerate(results):
            assert isinstance(result, ContentToolResult)
            assert result.value == f"Custom result {i + 1}"
            assert result.extra == {"index": i}
            assert result.request == request

    @pytest.mark.asyncio
    async def test_async_tool_yielding_multiple_results(self):
        """Test async tool function that yields multiple results."""
        chat = ChatOpenAI()

        async def async_multi_tool(count: int):
            """Async tool that yields multiple results."""
            for i in range(count):
                yield f"Async result {i + 1}"

        chat.register_tool(async_multi_tool)

        request = ContentToolRequest(
            id="test-id", name="async_multi_tool", arguments={"count": 2}
        )

        results = []
        async for result in chat._invoke_tool_async(request):
            results.append(result)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert isinstance(result, ContentToolResult)
            assert result.value == f"Async result {i + 1}"
            assert result.request == request
            assert result.error is None

    @pytest.mark.filterwarnings("ignore")
    def test_tool_yielding_with_error(self):
        """Test tool that yields some results then encounters an error."""
        chat = ChatOpenAI()

        def error_after_yield_tool(count: int):
            """Tool that yields some results then raises an error."""
            for i in range(count):
                if i == 2:
                    raise ValueError("Error after yielding 2 results")
                yield f"Result {i + 1}"

        chat.register_tool(error_after_yield_tool)

        request = ContentToolRequest(
            id="test-id", name="error_after_yield_tool", arguments={"count": 5}
        )

        results = list(chat._invoke_tool(request))

        # Should get 2 successful results + 1 error result
        assert len(results) == 3

        # First two should be successful
        assert results[0].value == "Result 1"
        assert results[0].error is None
        assert results[1].value == "Result 2"
        assert results[1].error is None

        # Third should be the error
        assert results[2].value is None
        assert results[2].error is not None
        assert "Error after yielding 2 results" in str(results[2].error)


class TestExistingToolsStillWork:
    """Test that existing tools continue to work with the changes."""

    def test_old_style_tool_invocation_still_works(self):
        """Test that tools registered the normal way still work correctly."""
        chat = ChatOpenAI()

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        chat.register_tool(add)

        request = ContentToolRequest(
            id="test-id", name="add", arguments={"x": 3, "y": 4}
        )

        results = list(chat._invoke_tool(request))
        assert len(results) == 1
        result = results[0]
        assert result.value == 7
        assert result.error is None

    @pytest.mark.filterwarnings("ignore")
    def test_unknown_tool_error_format_updated(self):
        """Test that unknown tool error message has been updated."""
        chat = ChatOpenAI()

        request = ContentToolRequest(
            id="test-id", name="nonexistent_tool", arguments={}
        )

        results = list(chat._invoke_tool(request))
        assert len(results) == 1
        result = results[0]
        assert result.value is None
        assert result.error is not None
        # The error message was updated to just "Unknown tool." instead of "Unknown tool: {name}"
        assert str(result.error) == "Unknown tool."

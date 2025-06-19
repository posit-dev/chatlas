"""Test Tool.from_mcp() class method."""

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from chatlas._content import ContentToolResultImage, ContentToolResultResource
from chatlas._tools import Tool
from chatlas.types import ContentToolResult
from pydantic.networks import AnyUrl

try:
    import mcp  # noqa: F401
except ImportError:
    pytest.skip("MCP package not available", allow_module_level=True)


class TestToolFromMCP:
    """Test Tool.from_mcp() class method."""

    def create_mock_mcp_tool(self, name: str, description: str, input_schema: dict):
        """Create a mock MCP tool."""
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.inputSchema = input_schema
        return tool

    def create_mock_session(self):
        """Create a mock MCP client session."""
        session = AsyncMock()
        return session

    def test_from_mcp_basic(self):
        """Test creating a Tool from basic MCP tool."""
        input_schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        }

        mcp_tool = self.create_mock_mcp_tool(
            name="add", description="Add two numbers", input_schema=input_schema
        )
        session = self.create_mock_session()

        tool = Tool.from_mcp(session, mcp_tool)

        assert tool.name == "add"

        func = tool.schema["function"]
        assert func["name"] == "add"
        assert func.get("description") == "Add two numbers"
        assert func.get("parameters") == {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
            "additionalProperties": False,
        }
        assert tool._is_async is True  # MCP tools are always async

    def test_from_mcp_empty_description(self):
        """Test creating a Tool from MCP tool with no description."""
        input_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }

        mcp_tool = self.create_mock_mcp_tool(
            name="test_tool", description="", input_schema=input_schema
        )
        session = self.create_mock_session()

        tool = Tool.from_mcp(session, mcp_tool)

        func = tool.schema["function"]
        assert func.get("description") == ""

    def test_from_mcp_complex_schema(self):
        """Test creating a Tool from MCP tool with complex input schema."""
        input_schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "options": {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                },
            },
            "required": ["items"],
        }

        mcp_tool = self.create_mock_mcp_tool(
            name="complex_tool", description="A complex tool", input_schema=input_schema
        )
        session = self.create_mock_session()

        tool = Tool.from_mcp(session, mcp_tool)

        expected_params = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "options": {
                    "type": "object",
                    "properties": {"flag": {"type": "boolean"}},
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        }

        func = tool.schema["function"]
        assert func.get("parameters") == expected_params

    @pytest.mark.asyncio
    async def test_mcp_tool_call_text_result(self):
        """Test calling an MCP tool that returns text content."""
        input_schema = {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

        mcp_tool = self.create_mock_mcp_tool(
            name="echo", description="Echo message", input_schema=input_schema
        )

        # Mock the session call_tool response
        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "text"
        mock_content.text = "Hello World"
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        # Call the tool function
        results = []
        async for result in await tool.func(message="Hello"):
            results.append(result)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ContentToolResult)
        assert result.value == "Hello World"

        # Verify session was called correctly
        session.call_tool.assert_called_once_with("echo", {"message": "Hello"})

    @pytest.mark.asyncio
    async def test_mcp_tool_call_image_result(self):
        """Test calling an MCP tool that returns image content."""
        mcp_tool = self.create_mock_mcp_tool(
            name="generate_image",
            description="Generate an image",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "image"
        mock_content.data = "base64imagedata"
        mock_content.mimeType = "image/png"
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        results = []
        async for result in await tool.func():
            results.append(result)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ContentToolResultImage)
        assert result.value == "base64imagedata"
        assert result.mime_type == "image/png"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_resource_result_text(self):
        """Test calling an MCP tool that returns text resource content."""
        from mcp.types import TextResourceContents

        mcp_tool = self.create_mock_mcp_tool(
            name="get_file",
            description="Get file contents",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "resource"

        # Mock text resource
        mock_resource = TextResourceContents(
            text="File contents here",
            uri=AnyUrl("file://path/to/file.txt"),
        )
        mock_resource.mimeType = "text/plain"
        mock_content.resource = mock_resource
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        results = []
        async for result in await tool.func():
            results.append(result)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ContentToolResultResource)
        assert result.value == b"File contents here"
        assert result.mime_type == "text/plain"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_resource_result_blob(self):
        """Test calling an MCP tool that returns blob resource content."""
        mcp_tool = self.create_mock_mcp_tool(
            name="get_binary",
            description="Get binary data",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "resource"

        # Mock blob resource
        mock_resource = MagicMock()
        mock_resource.blob = "binarydata"
        mock_resource.mimeType = "application/octet-stream"
        mock_content.resource = mock_resource
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        results = []
        async for result in await tool.func():
            results.append(result)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, ContentToolResultResource)
        assert result.value == b"binarydata"
        assert result.mime_type == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_multiple_results(self):
        """Test calling an MCP tool that returns multiple content items."""
        mcp_tool = self.create_mock_mcp_tool(
            name="multi_result",
            description="Return multiple results",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        # Multiple content items
        mock_content1 = MagicMock()
        mock_content1.type = "text"
        mock_content1.text = "First result"

        mock_content2 = MagicMock()
        mock_content2.type = "text"
        mock_content2.text = "Second result"

        mock_result.content = [mock_content1, mock_content2]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        results = []
        async for result in await tool.func():
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(r, ContentToolResult) for r in results)
        assert results[0].value == "First result"
        assert results[1].value == "Second result"

    @pytest.mark.asyncio
    async def test_mcp_tool_call_error(self):
        """Test calling an MCP tool that returns an error."""
        mcp_tool = self.create_mock_mcp_tool(
            name="error_tool",
            description="Tool that errors",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = True

        mock_content = MagicMock()
        mock_content.text = "Error executing tool error_tool: Something went wrong"
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        with pytest.raises(
            RuntimeError, match="Error executing tool error_tool: Something went wrong"
        ):
            async for result in await tool.func():
                pass

    @pytest.mark.asyncio
    async def test_mcp_tool_call_error_no_text_attribute(self):
        """Test calling an MCP tool that returns an error without text attribute."""
        mcp_tool = self.create_mock_mcp_tool(
            name="error_tool",
            description="Tool that errors",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = True

        mock_content = MagicMock()
        # No text attribute
        del mock_content.text
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        with pytest.raises(RuntimeError, match="Error executing tool error_tool"):
            async for result in await tool.func():
                pass

    @pytest.mark.asyncio
    async def test_mcp_tool_call_unsupported_image_type(self):
        """Test calling an MCP tool that returns unsupported image type."""
        mcp_tool = self.create_mock_mcp_tool(
            name="bad_image",
            description="Tool that returns unsupported image",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "image"
        mock_content.data = "imagedata"
        mock_content.mimeType = "image/unsupported"
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        with pytest.raises(
            ValueError, match="Unsupported image MIME type: image/unsupported"
        ):
            async for result in await tool.func():
                pass

    @pytest.mark.asyncio
    async def test_mcp_tool_call_unexpected_content_type(self):
        """Test calling an MCP tool that returns unexpected content type."""
        mcp_tool = self.create_mock_mcp_tool(
            name="bad_content",
            description="Tool with unexpected content",
            input_schema={"type": "object", "properties": {}},
        )

        session = self.create_mock_session()
        mock_result = MagicMock()
        mock_result.isError = False

        mock_content = MagicMock()
        mock_content.type = "unknown_type"
        mock_result.content = [mock_content]

        session.call_tool.return_value = mock_result

        tool = Tool.from_mcp(session, mcp_tool)

        with pytest.raises(RuntimeError, match="Unexpected content type: unknown_type"):
            async for result in await tool.func():
                pass

    def test_mcp_tool_input_schema_conversion(self):
        """Test that MCP tool input schema is properly converted."""
        # Test schema with title (should be removed)
        input_schema = {
            "type": "object",
            "title": "MySchema",
            "properties": {"param": {"type": "string", "title": "Parameter"}},
            "required": ["param"],
        }

        mcp_tool = self.create_mock_mcp_tool(
            name="test_tool", description="Test", input_schema=input_schema
        )
        session = self.create_mock_session()

        tool = Tool.from_mcp(session, mcp_tool)

        expected_params = {
            "type": "object",
            "properties": {"param": {"type": "string"}},
            "required": ["param"],
            "additionalProperties": False,
        }

        func = tool.schema["function"]
        assert func.get("parameters") == expected_params
        # Titles should be removed
        params = func.get("parameters", {})
        assert "title" not in params
        props = cast(dict, params.get("properties", {}))
        param = props.get("param", {})
        assert "title" not in param

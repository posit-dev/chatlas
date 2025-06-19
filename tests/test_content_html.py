"""Test HTML rendering improvements for ContentToolResult and related classes."""

from chatlas._content import ContentToolResultImage, ContentToolResultResource
from chatlas.types import ContentToolRequest, ContentToolResult


class TestContentToolRequestHTML:
    """Test HTML rendering for ContentToolRequest."""

    def test_content_tool_request_repr_html(self):
        """Test __repr_html__ method."""
        request = ContentToolRequest(
            id="test-id", name="test_tool", arguments={"x": 1, "y": 2}
        )

        # Should return string representation of tagify()
        html = request.__repr_html__()
        assert isinstance(html, str)
        assert "test_tool" in html


class TestContentToolResultHTML:
    """Test HTML rendering improvements for ContentToolResult."""

    def test_content_tool_result_repr_html(self):
        """Test __repr_html__ method."""
        # Create a request first
        request = ContentToolRequest(
            id="test-id", name="test_tool", arguments={"x": 1, "y": 2}
        )

        result = ContentToolResult(value="test result", request=request)

        # Should return string representation of tagify()
        html = result.__repr_html__()
        assert isinstance(html, str)
        assert "test_tool" in html

    def test_improved_html_structure(self):
        """Test the improved HTML structure with nested details."""

        request = ContentToolRequest(
            id="test-id",
            name="test_tool",
            arguments={"x": 1, "y": 2},
        )

        result = ContentToolResult(value="test result", request=request)

        html = result.tagify()
        html_str = str(html)

        # Should contain the new structure with nested details
        assert 'class="chatlas-tool-result"' in html_str
        assert "<details>" in html_str
        assert (
            "<summary>Result from tool call: <code>test_tool</code></summary>"
            in html_str
        )
        assert "<summary><strong>Result:</strong></summary>" in html_str
        assert "<summary><strong>Input parameters:</strong></summary>" in html_str

        # Should contain escaped content
        assert "test result" in html_str
        assert "1" in html_str
        assert "2" in html_str

    def test_html_with_error(self):
        """Test HTML rendering when there's an error."""

        request = ContentToolRequest(id="test-id", name="test_tool", arguments={"x": 1})

        result = ContentToolResult(
            value=None, error=ValueError("Test error"), request=request
        )

        html = result.tagify()
        html_str = str(html)

        # Should show error header
        assert "❌ Failed to call tool <code>test_tool</code>" in html_str
        assert "Tool call failed with error" in html_str

    def test_html_with_dict_arguments(self):
        """Test HTML rendering with dictionary arguments."""

        request = ContentToolRequest(
            id="test-id",
            name="test_tool",
            arguments={"param1": "value1", "param2": "value2"},
        )

        result = ContentToolResult(value="result", request=request)

        html = result.tagify()
        html_str = str(html)

        # Should have separate sections for each parameter
        assert "input-parameter-label" in html_str
        assert "param1" in html_str
        assert "param2" in html_str
        assert "value1" in html_str
        assert "value2" in html_str

    def test_html_with_non_dict_arguments(self):
        """Test HTML rendering with non-dictionary arguments."""

        request = ContentToolRequest(
            id="test-id", name="test_tool", arguments="simple string argument"
        )

        result = ContentToolResult(value="result", request=request)

        html = result.tagify()
        html_str = str(html)

        # Should contain the argument as a string
        assert "simple string argument" in html_str

    def test_html_escaping(self):
        """Test that HTML content is properly escaped."""

        request = ContentToolRequest(
            id="test-id",
            name="test_tool",
            arguments={"param": "<img src=x onerror=alert(1)>"},
        )

        result = ContentToolResult(
            value="<script>alert('xss')</script>", request=request
        )

        html = result.tagify()
        html_str = str(html)

        # HTML should be escaped
        assert "&lt;script&gt;" in html_str
        assert "&lt;img src=x" in html_str
        assert "<script>" not in html_str
        assert "<img" not in html_str

    def test_get_display_value_always_returns_string(self):
        """Test that _get_display_value always returns a string."""
        # Test with various value types
        test_cases = [
            ("string value", "string value"),
            (123, "123"),
            ([1, 2, 3], "[1, 2, 3]"),
            ({"key": "value"}, "{'key': 'value'}"),
            (None, "None"),
        ]

        for value, expected in test_cases:
            request = ContentToolRequest(id="test-id", name="test_tool", arguments={})
            result = ContentToolResult(value=value, request=request)
            display_value = result._get_display_value()
            assert isinstance(display_value, str)
            assert str(expected) == display_value

    def test_get_display_value_with_error(self):
        """Test _get_display_value when there's an error."""
        error = ValueError("Test error message")
        request = ContentToolRequest(id="test-id", name="test_tool", arguments={})
        result = ContentToolResult(value=None, error=error, request=request)

        display_value = result._get_display_value()
        assert isinstance(display_value, str)
        assert "Tool call failed with error: 'Test error message'" == display_value


class TestContentToolResultImageHTML:
    """Test HTML-related functionality for ContentToolResultImage."""

    def test_markdown_representation(self):
        """Test _repr_markdown_ method."""
        import base64

        image_data = base64.b64encode(b"fake image data").decode("utf-8")
        result = ContentToolResultImage(value=image_data, mime_type="image/png")

        markdown = result._repr_markdown_()
        expected = f"![](data:image/png;base64,{image_data})"
        assert markdown == expected


class TestContentToolResultResourceHTML:
    """Test HTML-related functionality for ContentToolResultResource."""

    def test_repr_mimebundle(self):
        """Test _repr_mimebundle_ method."""
        resource_data = b"This is some resource data"
        result = ContentToolResultResource(value=resource_data, mime_type="text/plain")

        mime_bundle = result._repr_mimebundle_()

        # Check the structure
        assert "text/plain" in mime_bundle
        assert mime_bundle["text/plain"] == "<text/plain object>"

    def test_repr_mimebundle_with_include_exclude(self):
        """Test _repr_mimebundle_ with include/exclude parameters."""
        resource_data = b"Test data"
        result = ContentToolResultResource(
            value=resource_data,
            mime_type="application/json",
        )

        # Test with include/exclude (they're not used in current implementation but method signature accepts them)
        mime_bundle1 = result._repr_mimebundle_(include=["application/json"])
        mime_bundle2 = result._repr_mimebundle_(exclude=["text/plain"])

        # Should return the same result regardless (current implementation ignores these params)
        assert mime_bundle1 == mime_bundle2
        assert "application/json" in mime_bundle1
